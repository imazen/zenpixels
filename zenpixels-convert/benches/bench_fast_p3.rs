use moxcms::{BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, TransformOptions};
use zenbench::prelude::*;
use zenpixels_convert::cms::ColorManagement;
use zenpixels_convert::{
    ColorPrimaries, ColorProfileSource, NamedProfile, PixelFormat, ZenCmsLite,
};

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

fn build_lite_xf(
    src: ColorProfileSource<'_>,
    dst: ColorProfileSource<'_>,
    fmt: PixelFormat,
) -> Box<dyn zenpixels_convert::cms::RowTransform> {
    ZenCmsLite::default()
        .build_source_transform(src, dst, fmt, fmt)
        .unwrap()
        .unwrap()
}

fn main() {
    let w: usize = 1920;
    let h: usize = 1080;
    let pixel_count = w * h;
    let data_f32 = make_test_data_f32_rgb(pixel_count);
    let data_u8 = make_test_data_u8_rgb(pixel_count);
    let f32_bytes = pixel_count * 3 * 4;
    let u8_bytes = pixel_count * 3;

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

    // Pre-build ZenCmsLite transforms
    let p3_srgb = |fmt| {
        build_lite_xf(
            ColorProfileSource::Named(NamedProfile::DisplayP3),
            ColorProfileSource::Named(NamedProfile::Srgb),
            fmt,
        )
    };
    let lite_p3_srgb_f32 = p3_srgb(PixelFormat::RgbF32);
    let lite_p3_srgb_u8 = p3_srgb(PixelFormat::Rgb8);
    let lite_bt2020_srgb_f32 = build_lite_xf(
        ColorProfileSource::Named(NamedProfile::Bt2020),
        ColorProfileSource::Named(NamedProfile::Srgb),
        PixelFormat::RgbF32,
    );
    let lite_bt2020pq_srgb_f32 = build_lite_xf(
        ColorProfileSource::Named(NamedProfile::Bt2020Pq),
        ColorProfileSource::Named(NamedProfile::Srgb),
        PixelFormat::RgbF32,
    );
    let lite_srgb_p3_f32 = build_lite_xf(
        ColorProfileSource::Named(NamedProfile::Srgb),
        ColorProfileSource::Named(NamedProfile::DisplayP3),
        PixelFormat::RgbF32,
    );

    let p3_to_srgb_matrix = ColorPrimaries::DisplayP3
        .gamut_matrix_to(ColorPrimaries::Bt709)
        .unwrap();

    zenbench::run(|suite| {
        // --- P3 → sRGB f32 ---
        suite.group("P3→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes(f32_bytes as u64));
            let src: Vec<u8> = bytemuck::cast_slice(&data_f32).to_vec();
            g.bench("ZenCmsLite", move |bench| {
                let mut dst = vec![0u8; src.len()];
                bench.iter(|| {
                    lite_p3_srgb_f32.transform_row(&src, &mut dst, w as u32);
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
            g.throughput(Throughput::Bytes(u8_bytes as u64));
            let d = data_u8.clone();
            g.bench("ZenCmsLite u8", move |bench| {
                let mut dst = vec![0u8; d.len()];
                bench.iter(|| {
                    lite_p3_srgb_u8.transform_row(&d, &mut dst, w as u32);
                    black_box(());
                });
            });
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

        // --- BT.2020 SDR → sRGB f32 ---
        suite.group("BT.2020 SDR→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes(f32_bytes as u64));
            let src: Vec<u8> = bytemuck::cast_slice(&data_f32).to_vec();
            g.bench("ZenCmsLite", move |bench| {
                let mut dst = vec![0u8; src.len()];
                bench.iter(|| {
                    lite_bt2020_srgb_f32.transform_row(&src, &mut dst, w as u32);
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

        // --- BT.2020 PQ → sRGB f32 ---
        suite.group("BT.2020 PQ→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes(f32_bytes as u64));
            let src: Vec<u8> = bytemuck::cast_slice(&data_f32).to_vec();
            g.bench("ZenCmsLite", move |bench| {
                let mut dst = vec![0u8; src.len()];
                bench.iter(|| {
                    lite_bt2020pq_srgb_f32.transform_row(&src, &mut dst, w as u32);
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

        // --- sRGB → P3 f32 ---
        suite.group("sRGB→P3 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes(f32_bytes as u64));
            let src: Vec<u8> = bytemuck::cast_slice(&data_f32).to_vec();
            g.bench("ZenCmsLite", move |bench| {
                let mut dst = vec![0u8; src.len()];
                bench.iter(|| {
                    lite_srgb_p3_f32.transform_row(&src, &mut dst, w as u32);
                    black_box(());
                });
            });
        });

        // --- Linear (matrix only, no TRC) ---
        suite.group("Linear P3→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes(f32_bytes as u64));
            let d = data_f32.clone();
            g.bench("matrix only (in-place)", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_gamut::convert_linear_rgb(&p3_to_srgb_matrix, &mut buf);
                    black_box(());
                });
            });
        });

        // --- f32→u8 encode comparison: LUT vs polynomial ---
        suite.group("f32→u8 sRGB encode 1080p RGB", |g| {
            // Throughput measured as output u8 bytes
            g.throughput(Throughput::Bytes(u8_bytes as u64));

            // Source: linear f32 values in [0,1]
            let linear_f32 = data_f32.clone();

            // LUT encode: linear_to_srgb_u8_slice (4096-entry const LUT)
            let d = linear_f32.clone();
            g.bench("LUT (4096-entry)", move |bench| {
                let mut dst = vec![0u8; d.len()];
                bench.iter(|| {
                    linear_srgb::default::linear_to_srgb_u8_slice(&d, &mut dst);
                    black_box(());
                });
            });

            // Polynomial encode: linear_to_srgb (rational poly) + quantize
            let d = linear_f32.clone();
            g.bench("polynomial + quantize", move |bench| {
                let mut dst = vec![0u8; d.len()];
                bench.iter(|| {
                    for (inp, out) in d.iter().zip(dst.iter_mut()) {
                        *out = (linear_srgb::tf::linear_to_srgb(*inp) * 255.0 + 0.5) as u8;
                    }
                    black_box(());
                });
            });

            // SIMD polynomial encode: linear_to_srgb f32→f32 slice then quantize
            let d = linear_f32.clone();
            g.bench("SIMD poly slice + quantize", move |bench| {
                let mut f32_buf = d.clone();
                let mut dst = vec![0u8; d.len()];
                bench.iter(|| {
                    // In-place f32 encode via SIMD polynomial
                    linear_srgb::default::linear_to_srgb_slice(&mut f32_buf);
                    // Then quantize f32→u8
                    for (f, out) in f32_buf.iter().zip(dst.iter_mut()) {
                        *out = (*f * 255.0 + 0.5) as u8;
                    }
                    // Reset for next iter
                    f32_buf.copy_from_slice(&d);
                    black_box(());
                });
            });
        });
    });
}
