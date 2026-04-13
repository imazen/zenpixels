use moxcms::{BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, TransformOptions};
use zenbench::prelude::*;

fn make_test_data_f32_rgb(pixels: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; pixels * 3];
    for i in 0..pixels {
        data[i * 3] = (i % 256) as f32 / 255.0;
        data[i * 3 + 1] = ((i / 256) % 256) as f32 / 255.0;
        data[i * 3 + 2] = ((i / 65536) % 256) as f32 / 255.0;
    }
    data
}

fn make_test_data_u8_rgb(pixels: usize) -> Vec<u8> {
    let mut data = vec![0u8; pixels * 3];
    for i in 0..pixels {
        data[i * 3] = (i % 256) as u8;
        data[i * 3 + 1] = ((i / 256) % 256) as u8;
        data[i * 3 + 2] = ((i / 65536) % 256) as u8;
    }
    data
}

fn moxcms_opts() -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

fn main() {
    let w: usize = 1920;
    let h: usize = 1080;
    let pixel_count = w * h;
    let data_f32 = make_test_data_f32_rgb(pixel_count);
    let data_u8 = make_test_data_u8_rgb(pixel_count);

    let opts = moxcms_opts();
    let p3 = ColorProfile::new_display_p3();
    let srgb = ColorProfile::new_srgb();
    let bt2020 = ColorProfile::new_bt2020();
    let bt2020_pq = ColorProfile::new_bt2020_pq();

    let xf_p3_srgb_f32 = p3
        .create_transform_f32(Layout::Rgb, &srgb, Layout::Rgb, opts)
        .unwrap();
    let xf_p3_srgb_u8 = p3
        .create_transform_8bit(Layout::Rgb, &srgb, Layout::Rgb, opts)
        .unwrap();
    let xf_bt2020_srgb_f32 = bt2020
        .create_transform_f32(Layout::Rgb, &srgb, Layout::Rgb, opts)
        .unwrap();
    let xf_bt2020pq_srgb_f32 = bt2020_pq
        .create_transform_f32(Layout::Rgb, &srgb, Layout::Rgb, opts)
        .unwrap();

    zenbench::run(|suite| {
        // --- P3 → sRGB ---
        suite.group("P3→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));
            let d = data_f32.clone();
            g.bench("fast_gamut", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_gamut::p3_to_srgb_f32(&mut buf);
                    black_box(());
                });
            });
            let d = data_f32.clone();
            let x = xf_p3_srgb_f32.clone();
            g.bench("moxcms", move |bench| {
                let mut dst = vec![0.0f32; d.len()];
                bench.iter(|| {
                    x.transform(&d, &mut dst).unwrap();
                    black_box(());
                });
            });
        });

        // --- P3 → sRGB u8 ---
        suite.group("P3→sRGB 1080p u8 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3) as u64));
            let d = data_u8.clone();
            g.bench("fast_gamut u8 (scalar)", move |bench| {
                let mut dst = vec![0u8; d.len()];
                bench.iter(|| {
                    zenpixels_convert::fast_gamut::p3_to_srgb_u8_rgb(&d, &mut dst);
                    black_box(());
                });
            });
            let d = data_u8.clone();
            {
                use zenpixels_convert::cms::ColorManagement;
                let cms = zenpixels_convert::ZenCmsLite;
                let src = zenpixels_convert::ColorProfileSource::Named(
                    zenpixels_convert::NamedProfile::DisplayP3,
                );
                let dst_profile = zenpixels_convert::ColorProfileSource::Named(
                    zenpixels_convert::NamedProfile::Srgb,
                );
                let xf = cms
                    .build_source_transform(
                        src,
                        dst_profile,
                        zenpixels_convert::PixelFormat::Rgb8,
                        zenpixels_convert::PixelFormat::Rgb8,
                    )
                    .unwrap()
                    .unwrap();
                g.bench("ZenCmsLite u8 (LUT)", move |bench| {
                    let mut dst = vec![0u8; d.len()];
                    bench.iter(|| {
                        xf.transform_row(&d, &mut dst, w as u32);
                        black_box(());
                    });
                });
            }
            let d = data_u8.clone();
            let x = xf_p3_srgb_u8.clone();
            g.bench("moxcms u8", move |bench| {
                let mut dst = vec![0u8; d.len()];
                bench.iter(|| {
                    x.transform(&d, &mut dst).unwrap();
                    black_box(());
                });
            });
        });

        // --- BT.2020 SDR → sRGB ---
        suite.group("BT.2020 SDR→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));
            let d = data_f32.clone();
            g.bench("fast_gamut", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_gamut::bt2020_sdr_to_srgb_f32(&mut buf);
                    black_box(());
                });
            });
            let d = data_f32.clone();
            let x = xf_bt2020_srgb_f32.clone();
            g.bench("moxcms", move |bench| {
                let mut dst = vec![0.0f32; d.len()];
                bench.iter(|| {
                    x.transform(&d, &mut dst).unwrap();
                    black_box(());
                });
            });
        });

        // --- BT.2020 PQ → sRGB ---
        suite.group("BT.2020 PQ→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));
            let d = data_f32.clone();
            g.bench("fast_gamut", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_gamut::bt2020_pq_to_srgb_f32(&mut buf);
                    black_box(());
                });
            });
            let d = data_f32.clone();
            let x = xf_bt2020pq_srgb_f32.clone();
            g.bench("moxcms", move |bench| {
                let mut dst = vec![0.0f32; d.len()];
                bench.iter(|| {
                    x.transform(&d, &mut dst).unwrap();
                    black_box(());
                });
            });
        });

        // --- sRGB → P3 ---
        suite.group("sRGB→P3 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));
            let d = data_f32.clone();
            g.bench("fast_gamut", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_gamut::srgb_to_p3_f32(&mut buf);
                    black_box(());
                });
            });
        });

        // --- Linear (matrix only) ---
        suite.group("Linear P3→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));
            let d = data_f32.clone();
            g.bench("fast_gamut linear", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_gamut::convert_linear_rgb(
                        &zenpixels_convert::fast_gamut::P3_TO_SRGB,
                        &mut buf,
                    );
                    black_box(());
                });
            });
        });
    });
}
