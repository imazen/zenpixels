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

/// Build a ZenCmsLite f32 RGB transform (src→dst copy-based via RowTransform).
fn build_lite_f32_xf(
    src: ColorProfileSource<'_>,
    dst: ColorProfileSource<'_>,
) -> Box<dyn zenpixels_convert::cms::RowTransform> {
    let cms = ZenCmsLite;
    cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32)
        .unwrap()
        .unwrap()
}

/// Build a ZenCmsLite u8 RGB transform.
fn build_lite_u8_xf(
    src: ColorProfileSource<'_>,
    dst: ColorProfileSource<'_>,
) -> Box<dyn zenpixels_convert::cms::RowTransform> {
    let cms = ZenCmsLite;
    cms.build_source_transform(src, dst, PixelFormat::Rgb8, PixelFormat::Rgb8)
        .unwrap()
        .unwrap()
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

    // Pre-build ZenCmsLite transforms
    let lite_p3_srgb_f32 = build_lite_f32_xf(
        ColorProfileSource::Named(NamedProfile::DisplayP3),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let lite_p3_srgb_u8_scalar = build_lite_u8_xf(
        ColorProfileSource::Named(NamedProfile::DisplayP3),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let lite_p3_srgb_u8_lut = build_lite_u8_xf(
        ColorProfileSource::Named(NamedProfile::DisplayP3),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let lite_bt2020_srgb_f32 = build_lite_f32_xf(
        ColorProfileSource::Named(NamedProfile::Bt2020),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let lite_bt2020pq_srgb_f32 = build_lite_f32_xf(
        ColorProfileSource::Named(NamedProfile::Bt2020Pq),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let lite_srgb_p3_f32 = build_lite_f32_xf(
        ColorProfileSource::Named(NamedProfile::Srgb),
        ColorProfileSource::Named(NamedProfile::DisplayP3),
    );

    let p3_to_srgb_matrix = ColorPrimaries::DisplayP3
        .gamut_matrix_to(ColorPrimaries::Bt709)
        .unwrap();

    zenbench::run(|suite| {
        // --- P3 → sRGB ---
        suite.group("P3→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));
            let d = data_f32.clone();
            g.bench("fast_gamut", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    let src_copy: Vec<u8> = bytemuck::cast_slice(&buf).to_vec();
                    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                    lite_p3_srgb_f32.transform_row(&src_copy, dst_bytes, w as u32);
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
                    lite_p3_srgb_u8_scalar.transform_row(&d, &mut dst, w as u32);
                    black_box(());
                });
            });
            let d = data_u8.clone();
            g.bench("ZenCmsLite u8 (LUT)", move |bench| {
                let mut dst = vec![0u8; d.len()];
                bench.iter(|| {
                    lite_p3_srgb_u8_lut.transform_row(&d, &mut dst, w as u32);
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

        // --- BT.2020 SDR → sRGB ---
        suite.group("BT.2020 SDR→sRGB 1080p f32 RGB", |g| {
            g.throughput(Throughput::Bytes((pixel_count * 3 * 4) as u64));
            let d = data_f32.clone();
            g.bench("fast_gamut", move |bench| {
                let mut buf = d.clone();
                bench.iter(|| {
                    let src_copy: Vec<u8> = bytemuck::cast_slice(&buf).to_vec();
                    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                    lite_bt2020_srgb_f32.transform_row(&src_copy, dst_bytes, w as u32);
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
                    let src_copy: Vec<u8> = bytemuck::cast_slice(&buf).to_vec();
                    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                    lite_bt2020pq_srgb_f32.transform_row(&src_copy, dst_bytes, w as u32);
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
                    let src_copy: Vec<u8> = bytemuck::cast_slice(&buf).to_vec();
                    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                    lite_srgb_p3_f32.transform_row(&src_copy, dst_bytes, w as u32);
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
                    zenpixels_convert::fast_gamut::convert_linear_rgb(&p3_to_srgb_matrix, &mut buf);
                    black_box(());
                });
            });
        });
    });
}
