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

    // Standalone pass benchmarks to identify bottleneck
    let data_linear_rgb = {
        let mut d = data_rgb.clone();
        linear_srgb::default::srgb_to_linear_slice(&mut d);
        d
    };

    zenbench::run(|suite| {
        suite.group("Pass breakdown  1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));

            let d = data_rgb.clone();
            g.bench("linearize only", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    linear_srgb::default::srgb_to_linear_slice(&mut buf);
                    black_box(());
                });
            });

            let d = data_linear_rgb.clone();
            g.bench("matrix stride-3 scalar", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    for pixel in buf.chunks_exact_mut(3) {
                        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
                        pixel[0] =
                            1.2249401763_f32.mul_add(r, (-0.2249401763_f32).mul_add(g, 0.0 * b));
                        pixel[1] =
                            (-0.0420569547_f32).mul_add(r, 1.0420569547_f32.mul_add(g, 0.0 * b));
                        pixel[2] = (-0.0196375546_f32)
                            .mul_add(r, (-0.0786360456_f32).mul_add(g, 1.0982736001_f32 * b));
                    }
                    black_box(());
                });
            });

            // RGBX approach: pad to stride-4, matrix on stride-4, strip padding
            let d = data_linear_rgb.clone();
            g.bench("matrix via RGBX pad+strip", move |bench| {
                let mut buf = d.clone();
                let n = buf.len() / 3;
                let mut rgbx = vec![0.0f32; n * 4];
                bench.iter(|| {
                    // Pad RGB → RGBX
                    for i in 0..n {
                        rgbx[i * 4] = buf[i * 3];
                        rgbx[i * 4 + 1] = buf[i * 3 + 1];
                        rgbx[i * 4 + 2] = buf[i * 3 + 2];
                        // rgbx[i*4+3] = 0.0; // X channel ignored
                    }
                    // Matrix on stride-4
                    for pixel in rgbx.chunks_exact_mut(4) {
                        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
                        pixel[0] =
                            1.2249401763_f32.mul_add(r, (-0.2249401763_f32).mul_add(g, 0.0 * b));
                        pixel[1] =
                            (-0.0420569547_f32).mul_add(r, 1.0420569547_f32.mul_add(g, 0.0 * b));
                        pixel[2] = (-0.0196375546_f32)
                            .mul_add(r, (-0.0786360456_f32).mul_add(g, 1.0982736001_f32 * b));
                    }
                    // Strip RGBX → RGB
                    for i in 0..n {
                        buf[i * 3] = rgbx[i * 4];
                        buf[i * 3 + 1] = rgbx[i * 4 + 1];
                        buf[i * 3 + 2] = rgbx[i * 4 + 2];
                    }
                    black_box(());
                });
            });

            // Planar approach: separate loops per output channel
            let d = data_linear_rgb.clone();
            g.bench("matrix planar separate loops", move |bench| {
                let mut buf = d.clone();
                let n = buf.len() / 3;
                let mut pr = vec![0.0f32; n];
                let mut pg = vec![0.0f32; n];
                let mut pb = vec![0.0f32; n];
                bench.iter(|| {
                    // Deinterleave
                    for i in 0..n {
                        pr[i] = buf[i * 3];
                        pg[i] = buf[i * 3 + 1];
                        pb[i] = buf[i * 3 + 2];
                    }
                    // Separate loop per output channel — maximizes auto-vectorization
                    let mut or = vec![0.0f32; n];
                    let mut og = vec![0.0f32; n];
                    let mut ob = vec![0.0f32; n];
                    for i in 0..n {
                        or[i] = 1.2249401763_f32.mul_add(pr[i], (-0.2249401763_f32) * pg[i]);
                    }
                    for i in 0..n {
                        og[i] = (-0.0420569547_f32).mul_add(pr[i], 1.0420569547_f32 * pg[i]);
                    }
                    for i in 0..n {
                        ob[i] = (-0.0196375546_f32).mul_add(
                            pr[i],
                            (-0.0786360456_f32).mul_add(pg[i], 1.0982736001_f32 * pb[i]),
                        );
                    }
                    // Interleave
                    for i in 0..n {
                        buf[i * 3] = or[i];
                        buf[i * 3 + 1] = og[i];
                        buf[i * 3 + 2] = ob[i];
                    }
                    black_box(());
                });
            });

            let d = data_linear_rgb.clone();
            g.bench("encode only", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    linear_srgb::default::linear_to_srgb_slice(&mut buf);
                    black_box(());
                });
            });
        });
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
