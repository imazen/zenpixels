//! `Cmyk8` to sRGB through `MoxCms`, against the same profile at F32 and the path `Cmyk8` used to take.
//!
//! `old 8-bit path` is that code, restated here with moxcms directly: its F32 transform with no
//! black-point compensation, rounded to 8 bits. Rows are 4096 pixels. The synthetic profile always
//! runs; `ZENPIXELS_CMYK_ICC=<path>` adds a real one.
//!
//! Run: `cargo bench -p zenpixels-convert --features cms-moxcms --bench bench_cmyk8 -- --no-busy-gate`

use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent,
    TransformExecutor, TransformOptions,
};
use std::sync::Arc;
use zenbench::prelude::*;
use zenpixels::{Cicp, ColorProfileSource as Source, PixelFormat as F};
use zenpixels_convert::cms::{PluggableCms, RowTransformMut};
use zenpixels_convert::cms_moxcms::MoxCms;
use zenpixels_convert::policy::ConvertOptions;

#[path = "../tests/common/synthetic_icc.rs"]
#[allow(dead_code)]
mod synthetic_icc;

const WIDTH: usize = 4096;
const SRGB: Source<'static> = Source::Cicp(Cicp::new(1, 13, 0, true));

fn transform(icc: &[u8], from: F, to: F) -> Box<dyn RowTransformMut> {
    MoxCms
        .build_source_transform(
            Source::Icc(icc),
            SRGB,
            from,
            to,
            &ConvertOptions::permissive(),
        )
        .expect("declined")
        .expect("refused")
}

fn old_path(icc: &[u8]) -> Arc<dyn TransformExecutor<f32> + Send + Sync> {
    let source = ColorProfile::new_from_slice(icc).expect("profile");
    let target = ColorProfile::new_srgb();
    let intent = if source.lut_a_to_b_colorimetric.is_none() {
        RenderingIntent::Perceptual
    } else {
        RenderingIntent::RelativeColorimetric
    };
    #[allow(clippy::needless_update)]
    let options = TransformOptions {
        rendering_intent: intent,
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        prefer_fixed_point: false,
        allow_extended_range_rgb_xyz: true,
        // Feature unification can add fields to this struct.
        ..Default::default()
    };
    source
        .create_transform_f32(Layout::Rgba, &target, Layout::Rgb, options)
        .expect("transform")
}

fn main() {
    let mut profiles = vec![("synthetic".to_string(), synthetic_icc::synthetic_cmyk_icc())];
    if let Ok(path) = std::env::var("ZENPIXELS_CMYK_ICC") {
        let name = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        profiles.push((name, std::fs::read(&path).expect("profile")));
    }
    let ink: Vec<u8> = (0..WIDTH as u32 * 4)
        .map(|i| ((i * 37 + i / 4 * 11) % 256) as u8)
        .collect();
    let ink_f32: Vec<f32> = ink.iter().map(|&v| f32::from(v) / 255.0).collect();

    zenbench::run(move |suite| {
        for (name, icc) in &profiles {
            let (icc, ink, ink_f32) = (icc.clone(), ink.clone(), ink_f32.clone());
            suite.group(format!("{name} CMYK to sRGB, 4096 pixels"), move |g| {
                g.throughput(Throughput::Elements(WIDTH as u64));
                {
                    let mut t = transform(&icc, F::Cmyk8, F::Rgb8);
                    let (src, mut dst) = (ink.clone(), vec![0u8; WIDTH * 3]);
                    g.bench("Cmyk8 to Rgb8", move |b| {
                        b.iter(|| {
                            t.try_transform_row(&src, &mut dst, WIDTH as u32).unwrap();
                            black_box(dst[0])
                        })
                    });
                }
                {
                    let mut t = transform(&icc, F::CmykF32, F::RgbF32);
                    let src = ink_f32.clone();
                    let mut dst = vec![0f32; WIDTH * 3];
                    g.bench("CmykF32 to RgbF32", move |b| {
                        b.iter(|| {
                            t.try_transform_row(
                                bytemuck::cast_slice(&src),
                                bytemuck::cast_slice_mut(&mut dst),
                                WIDTH as u32,
                            )
                            .unwrap();
                            black_box(dst[0])
                        })
                    });
                }
                {
                    let t = old_path(&icc);
                    let src = ink_f32.clone();
                    let mut float = vec![0f32; WIDTH * 3];
                    let mut dst = vec![0u8; WIDTH * 3];
                    g.bench("old 8-bit path", move |b| {
                        b.iter(|| {
                            t.transform(&src, &mut float).unwrap();
                            for (byte, v) in dst.iter_mut().zip(&float) {
                                *byte = (f64::from(v.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                            }
                            black_box(dst[0])
                        })
                    });
                }
            });
        }
    });
}
