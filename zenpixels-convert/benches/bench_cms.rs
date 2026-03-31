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
    // create_transform_8bit always uses the CLut/LUT path — exercises weight scale
    src.create_transform_8bit(Layout::Rgba, &dst, Layout::Rgba, opts).unwrap()
}

/// Standard options used in production (fixed-point trilinear, High weight scale).
fn opts_standard() -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        ..Default::default()
    }
}

/// Float tetrahedral with High weight scale — higher quality, unknown perf cost.
fn opts_tetrahedral_float() -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        prefer_fixed_point: false,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

fn main() {
    // 4K RGBA
    let w: usize = 3840;
    let h: usize = 2160;
    let pixels = w * h * 4;

    let t_low = make_lut_transform(TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::Low,
        ..Default::default()
    });
    let t_standard = make_lut_transform(opts_standard());
    let t_tetra = make_lut_transform(opts_tetrahedral_float());

    zenbench::run(|suite| {
        suite.group("moxcms ProPhoto→sRGB LUT  4K (3840×2160)", |g| {
            g.throughput(Throughput::Bytes(pixels as u64));

            let src: Vec<u8> = (0..pixels).map(|i| (i % 256) as u8).collect();
            let mut dst_low = vec![0u8; pixels];
            let mut dst_std = vec![0u8; pixels];
            let mut dst_tetra = vec![0u8; pixels];

            g.bench("Low  fixed-point trilinear (old default)", {
                let s = src.clone();
                move |bench| bench.iter(|| {
                    t_low.transform(&s, &mut dst_low).unwrap();
                    black_box(());
                })
            });

            g.bench("High fixed-point trilinear (current standard)", {
                let s = src.clone();
                move |bench| bench.iter(|| {
                    t_standard.transform(&s, &mut dst_std).unwrap();
                    black_box(());
                })
            });

            g.bench("High float tetrahedral", {
                let s = src.clone();
                move |bench| bench.iter(|| {
                    t_tetra.transform(&s, &mut dst_tetra).unwrap();
                    black_box(());
                })
            });
        });
    });
}
