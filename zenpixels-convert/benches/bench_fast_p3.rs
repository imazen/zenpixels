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

fn make_test_data_f32_rgba(pixels: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; pixels * 4];
    for i in 0..pixels {
        data[i * 4] = (i % 256) as f32 / 255.0;
        data[i * 4 + 1] = ((i / 256) % 256) as f32 / 255.0;
        data[i * 4 + 2] = ((i / 65536) % 256) as f32 / 255.0;
        data[i * 4 + 3] = 1.0;
    }
    data
}

fn main() {
    let w: usize = 1920;
    let h: usize = 1080;
    let pixel_count = w * h;
    let data_rgb = make_test_data_f32_rgb(pixel_count);
    let data_rgba = make_test_data_f32_rgba(pixel_count);

    let opts = TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    };

    let src_p3 = ColorProfile::new_display_p3();
    let dst_srgb = ColorProfile::new_srgb();
    let src_srgb = ColorProfile::new_srgb();
    let dst_p3 = ColorProfile::new_display_p3();

    let xform_p3_srgb_rgb = src_p3
        .create_transform_f32(Layout::Rgb, &dst_srgb, Layout::Rgb, opts)
        .unwrap();
    let xform_p3_srgb_rgba = src_p3
        .create_transform_f32(Layout::Rgba, &dst_srgb, Layout::Rgba, opts)
        .unwrap();
    let xform_srgb_p3_rgb = src_srgb
        .create_transform_f32(Layout::Rgb, &dst_p3, Layout::Rgb, opts)
        .unwrap();

    zenbench::run(|suite| {
        suite.group("P3→sRGB  1080p (1920×1080)  f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));

            let d = data_rgb.clone();
            g.bench("fast_p3 (in-place)", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_p3::p3_to_srgb_f32(&mut buf);
                    black_box(());
                });
            });

            let d = data_rgb.clone();
            let x = xform_p3_srgb_rgb.clone();
            g.bench("moxcms (copy)", move |bench| {
                let mut dst = vec![0.0f32; d.len()];
                bench.iter(|| {
                    x.transform(&d, &mut dst).unwrap();
                    black_box(());
                });
            });
        });

        suite.group("P3→sRGB  1080p (1920×1080)  f32 RGBA", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 4 * 4) as u64));

            let d = data_rgba.clone();
            g.bench("fast_p3 (in-place)", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_p3::p3_to_srgb_f32_rgba(&mut buf);
                    black_box(());
                });
            });

            let d = data_rgba.clone();
            let x = xform_p3_srgb_rgba.clone();
            g.bench("moxcms (copy)", move |bench| {
                let mut dst = vec![0.0f32; d.len()];
                bench.iter(|| {
                    x.transform(&d, &mut dst).unwrap();
                    black_box(());
                });
            });
        });

        suite.group("sRGB→P3  1080p (1920×1080)  f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));

            let d = data_rgb.clone();
            g.bench("fast_p3 (in-place)", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    zenpixels_convert::fast_p3::srgb_to_p3_f32(&mut buf);
                    black_box(());
                });
            });

            let d = data_rgb.clone();
            let x = xform_srgb_p3_rgb.clone();
            g.bench("moxcms (copy)", move |bench| {
                let mut dst = vec![0.0f32; d.len()];
                bench.iter(|| {
                    x.transform(&d, &mut dst).unwrap();
                    black_box(());
                });
            });
        });
    });
}
