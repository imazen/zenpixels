use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, Transform8BitExecutor,
    TransformOptions,
};
use std::sync::Arc;
use zenbench::prelude::*;

fn make_lut_transform(opts: TransformOptions) -> Arc<Transform8BitExecutor> {
    // ProPhoto has a very wide gamut — forces the LUT interpolation path (non-in-place)
    // when converted to sRGB, exercising BarycentricWeightScale and InterpolationMethod.
    let src = ColorProfile::new_pro_photo_rgb();
    let dst = ColorProfile::new_srgb();
    // create_transform_8bit always uses the CLut/LUT path
    src.create_transform_8bit(Layout::Rgba, &dst, Layout::Rgba, opts).unwrap()
}

fn opts_fixed(method: InterpolationMethod) -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        prefer_fixed_point: true,
        interpolation_method: method,
        ..Default::default()
    }
}

fn opts_float(method: InterpolationMethod) -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        prefer_fixed_point: false,
        interpolation_method: method,
        ..Default::default()
    }
}

fn main() {
    // 4K RGBA
    let w: usize = 3840;
    let h: usize = 2160;
    let pixels = w * h * 4;

    let transforms: &[(&str, Arc<Transform8BitExecutor>)] = &[
        ("fixed  Linear",                make_lut_transform(opts_fixed(InterpolationMethod::Linear))),
        ("fixed  Tetrahedral",           make_lut_transform(opts_fixed(InterpolationMethod::Tetrahedral))),
        ("fixed  Pyramid",               make_lut_transform(opts_fixed(InterpolationMethod::Pyramid))),
        ("fixed  Prism",                 make_lut_transform(opts_fixed(InterpolationMethod::Prism))),
        ("float  Linear",                make_lut_transform(opts_float(InterpolationMethod::Linear))),
        ("float  Tetrahedral",           make_lut_transform(opts_float(InterpolationMethod::Tetrahedral))),
        ("float  Pyramid",               make_lut_transform(opts_float(InterpolationMethod::Pyramid))),
        ("float  Prism",                 make_lut_transform(opts_float(InterpolationMethod::Prism))),
    ];

    let src: Vec<u8> = (0..pixels).map(|i| (i % 256) as u8).collect();

    zenbench::run(|suite| {
        suite.group("moxcms ProPhoto→sRGB LUT  4K (3840×2160)  High weight scale", |g| {
            g.throughput(Throughput::Bytes(pixels as u64));

            for (name, t) in transforms {
                let mut dst = vec![0u8; pixels];
                let s = src.clone();
                let t = t.clone();
                g.bench(*name, move |bench| bench.iter(|| {
                    t.transform(&s, &mut dst).unwrap();
                    black_box(());
                }));
            }
        });
    });
}
