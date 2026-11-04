//! BGRA kernels exist for 8 bits only. Every plan that touches BGRA must run them at `U8`.
use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as T, PixelDescriptor, PixelFormat,
    TransferFunction as Tf,
};
use zenpixels_convert::{AlphaPolicy, ConvertOptions, ConvertPlan, RowConverter};

const TYPES: [T; 4] = [T::U8, T::U16, T::F16, T::F32];
const LAYOUTS: [L; 9] = [
    L::Rgb,
    L::Rgba,
    L::Bgra,
    L::Gray,
    L::GrayAlpha,
    L::Oklab,
    L::OklabA,
    L::Bgr,
    L::Cmyk,
];
const ALPHAS: [Option<AlphaMode>; 4] = [
    None,
    Some(AlphaMode::Straight),
    Some(AlphaMode::Premultiplied),
    Some(AlphaMode::Opaque),
];
const TRANSFERS: [Tf; 5] = [Tf::Srgb, Tf::Linear, Tf::Bt709, Tf::Pq, Tf::Hlg];

fn descriptors() -> Vec<PixelDescriptor> {
    let mut all = Vec::new();
    for t in TYPES {
        for l in LAYOUTS {
            for a in ALPHAS {
                if (a.is_some() != l.has_alpha()) || PixelFormat::from_parts(t, l, a).is_none() {
                    continue;
                }
                for tf in TRANSFERS {
                    all.push(PixelDescriptor::new(t, l, a, tf));
                }
            }
        }
    }
    all
}

#[test]
fn no_format_pair_runs_a_bgra_step_above_8_bits() {
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    let all = descriptors();
    assert!(all.iter().any(|d| d.layout() == L::Bgra));
    let (mut ran, mut bgra) = (0u32, 0u32);
    let src = [0x40u8; 64];
    let mut dst = [0u8; 64];
    for &from in &all {
        for &to in &all {
            let Ok(mut c) = RowConverter::new_explicit(from, to, &options) else {
                continue;
            };
            let (n, m) = (2 * from.bytes_per_pixel(), 2 * to.bytes_per_pixel());
            c.convert_row(&src[..n], &mut dst[..m], 2);
            ran += 1;
            if from.layout() == L::Bgra || to.layout() == L::Bgra {
                bgra += 1;
            }
        }
    }
    assert!(ran > 1000, "ran {ran}");
    assert!(bgra > 100, "bgra {bgra}");
}

#[test]
fn composed_plans_keep_bgra_steps_at_8_bits() {
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    let all = descriptors();
    let bgra: Vec<_> = all
        .iter()
        .copied()
        .filter(|d| d.layout() == L::Bgra)
        .collect();
    let mut composed = 0u32;
    let src = [0x40u8; 64];
    let mut dst = [0u8; 64];
    let plan = |a, b| ConvertPlan::new_explicit(a, b, &options).ok();
    for &b in &bgra {
        for &other in &all {
            for (x, y, z) in [(b, other, b), (other, b, other)] {
                let (Some(p), Some(q)) = (plan(x, y), plan(y, z)) else {
                    continue;
                };
                let Some(joined) = p.compose(&q) else {
                    continue;
                };
                let mut c = RowConverter::from_plan(joined);
                let (n, m) = (2 * x.bytes_per_pixel(), 2 * z.bytes_per_pixel());
                c.convert_row(&src[..n], &mut dst[..m], 2);
                composed += 1;
            }
        }
    }
    assert!(composed > 100, "composed {composed}");
}
