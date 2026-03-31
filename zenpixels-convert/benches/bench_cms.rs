use moxcms::{BarycentricWeightScale, ColorProfile, Layout, Transform8BitExecutor, TransformOptions};
use std::sync::Arc;
use zenbench::prelude::*;

fn make_lut_transform(scale: BarycentricWeightScale) -> Arc<Transform8BitExecutor> {
    // ProPhoto has a very wide gamut — forces the LUT interpolation path (non-in-place)
    // when converted to sRGB, exercising BarycentricWeightScale.
    let src = ColorProfile::new_pro_photo_rgb();
    let dst = ColorProfile::new_srgb();
    let opts = TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: scale,
        ..Default::default()
    };
    // create_transform_8bit always uses the CLut/LUT path — exercises weight scale
    src.create_transform_8bit(Layout::Rgba, &dst, Layout::Rgba, opts).unwrap()
}

fn main() {
    // 4K RGBA
    let w: usize = 3840;
    let h: usize = 2160;
    let pixels = w * h * 4;

    let t_low = make_lut_transform(BarycentricWeightScale::Low);
    let t_high = make_lut_transform(BarycentricWeightScale::High);

    zenbench::run(|suite| {
        suite.group("moxcms ProPhoto→sRGB LUT  4K (3840×2160)", |g| {
            g.throughput(Throughput::Bytes(pixels as u64));

            let src: Vec<u8> = (0..pixels).map(|i| (i % 256) as u8).collect();
            let mut dst_low = vec![0u8; pixels];
            let mut dst_high = vec![0u8; pixels];

            g.bench("Low  (default)", {
                let s = src.clone();
                move |bench| bench.iter(|| {
                    t_low.transform(&s, &mut dst_low).unwrap();
                    black_box(());
                })
            });

            g.bench("High (options)", {
                let s = src.clone();
                move |bench| bench.iter(|| {
                    t_high.transform(&s, &mut dst_high).unwrap();
                    black_box(());
                })
            });
        });
    });
}
