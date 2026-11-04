//! The matte blend against the formula, sample by sample, at every depth and transfer.
use linear_srgb::{default as ds, tf};
use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as T, PixelDescriptor as D, TransferFunction as Tf,
};
use zenpixels_convert::{AlphaPolicy, ConvertOptions, RowConverter};

#[allow(clippy::excessive_precision)]
const GAMMA: f32 = 2.19921875;
const MATTE: (u8, u8, u8) = (200, 100, 50);
const WIDTH: usize = 4099;

fn eotf(t: Tf, v: f32) -> f32 {
    match t {
        Tf::Srgb => tf::srgb_to_linear(v),
        Tf::Bt709 => tf::bt709_to_linear(v),
        Tf::Pq => tf::pq_to_linear(v),
        Tf::Hlg => tf::hlg_to_linear(v),
        Tf::Gamma22 => ds::gamma_to_linear(v, GAMMA),
        _ => v,
    }
}

fn oetf(t: Tf, v: f32) -> f32 {
    match t {
        Tf::Srgb => tf::linear_to_srgb(v),
        Tf::Bt709 => tf::linear_to_bt709(v),
        Tf::Pq => tf::linear_to_pq(v),
        Tf::Hlg => tf::linear_to_hlg(v),
        Tf::Gamma22 => ds::linear_to_gamma(v, GAMMA),
        _ => v,
    }
}

fn blend(v: f32, a: f32, m: f32) -> f32 {
    v * a.clamp(0.0, 1.0) + m * (1.0 - a.clamp(0.0, 1.0))
}

fn matte() -> [f32; 3] {
    [MATTE.0, MATTE.1, MATTE.2].map(ds::srgb_u8_to_linear)
}

fn converter(t: T, transfer: Tf) -> RowConverter {
    let from = D::new(t, L::Rgba, Some(AlphaMode::Straight), transfer);
    let to = D::new(t, L::Rgb, None, transfer);
    let policy = AlphaPolicy::CompositeOnto {
        r: MATTE.0,
        g: MATTE.1,
        b: MATTE.2,
    };
    RowConverter::new_explicit(
        from,
        to,
        &ConvertOptions::permissive().with_alpha_policy(policy),
    )
    .unwrap()
}

fn samples(max: u32) -> Vec<u32> {
    let mut x = 0x2545_F491u32;
    (0..WIDTH * 4)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x % (max + 1)
        })
        .collect()
}

const CURVES: [Tf; 6] = [
    Tf::Srgb,
    Tf::Linear,
    Tf::Bt709,
    Tf::Pq,
    Tf::Hlg,
    Tf::Gamma22,
];

/// sRGB U8 uses the same tables as before: no difference at all.
#[test]
fn srgb_u8_matches_its_tables_exactly() {
    let m = matte();
    let src: Vec<u8> = samples(255).iter().map(|v| *v as u8).collect();
    let mut out = vec![0u8; WIDTH * 3];
    converter(T::U8, Tf::Srgb).convert_row(&src, &mut out, WIDTH as u32);
    for (i, p) in src.as_chunks::<4>().0.iter().enumerate() {
        let a = f32::from(p[3]) * (1.0 / 255.0);
        for k in 0..3 {
            let lin = blend(ds::srgb_u8_to_linear(p[k]), a, m[k]);
            assert_eq!(out[i * 3 + k], ds::linear_to_srgb_u8(lin), "pixel {p:?}");
        }
    }
}

#[test]
fn integer_output_stays_within_one_code() {
    let m = matte();
    for transfer in CURVES {
        if transfer != Tf::Srgb {
            let src: Vec<u8> = samples(255).iter().map(|v| *v as u8).collect();
            let mut out = vec![0u8; WIDTH * 3];
            converter(T::U8, transfer).convert_row(&src, &mut out, WIDTH as u32);
            for (i, p) in src.as_chunks::<4>().0.iter().enumerate() {
                let a = f32::from(p[3]) * (1.0 / 255.0);
                for k in 0..3 {
                    let v = eotf(transfer, f32::from(p[k]) * (1.0 / 255.0));
                    let want = oetf(transfer, blend(v, a, m[k])).clamp(0.0, 1.0) * 255.0 + 0.5;
                    assert!(
                        (i32::from(out[i * 3 + k]) - want as i32).abs() <= 1,
                        "{transfer:?} {p:?}"
                    );
                }
            }
        }
        let src: Vec<u16> = samples(65535).iter().map(|v| *v as u16).collect();
        let mut out = vec![0u16; WIDTH * 3];
        converter(T::U16, transfer).convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut out),
            WIDTH as u32,
        );
        for (i, p) in src.as_chunks::<4>().0.iter().enumerate() {
            let a = f32::from(p[3]) * (1.0 / 65535.0);
            for k in 0..3 {
                let v = eotf(transfer, f32::from(p[k]) * (1.0 / 65535.0));
                let want = oetf(transfer, blend(v, a, m[k])).clamp(0.0, 1.0) * 65535.0 + 0.5;
                assert!(
                    (i32::from(out[i * 3 + k]) - want as i32).abs() <= 1,
                    "{transfer:?} {p:?}"
                );
            }
        }
    }
}

#[test]
fn float_output_stays_within_rounding() {
    let m = matte();
    for transfer in CURVES {
        let src: Vec<f32> = samples(1 << 24)
            .iter()
            .map(|v| *v as f32 / (1 << 24) as f32)
            .collect();
        let mut out = vec![0f32; WIDTH * 3];
        converter(T::F32, transfer).convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut out),
            WIDTH as u32,
        );
        for (i, p) in src.as_chunks::<4>().0.iter().enumerate() {
            for k in 0..3 {
                let want = oetf(transfer, blend(eotf(transfer, p[k]), p[3], m[k]));
                let got = out[i * 3 + k];
                assert!(
                    (got - want).abs() <= 1e-6,
                    "{transfer:?} {p:?}: {got} {want}"
                );
            }
        }
    }
}

#[test]
fn half_float_output_stays_within_one_step() {
    let m = matte();
    for transfer in CURVES {
        let src: Vec<u16> = samples(1 << 24)
            .iter()
            .map(|v| half::f16::from_f32(*v as f32 / (1 << 24) as f32).to_bits())
            .collect();
        let mut out = vec![0u16; WIDTH * 3];
        converter(T::F16, transfer).convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut out),
            WIDTH as u32,
        );
        for (i, p) in src.as_chunks::<4>().0.iter().enumerate() {
            let a = half::f16::from_bits(p[3]).to_f32();
            for k in 0..3 {
                let v = eotf(transfer, half::f16::from_bits(p[k]).to_f32());
                let want = half::f16::from_f32(oetf(transfer, blend(v, a, m[k]))).to_bits();
                assert!(
                    out[i * 3 + k].abs_diff(want) <= 1,
                    "{transfer:?} {p:?}: {} {want}",
                    out[i * 3 + k]
                );
            }
        }
    }
}

/// Values outside `[0, 1]` behave like the formula, or the difference is on record here.
#[test]
fn out_of_range_float_samples_follow_the_formula() {
    let m = matte();
    for transfer in CURVES {
        let mut src = Vec::new();
        for v in [-0.5f32, -0.01, 1.01, 1.5, 4.0] {
            for a in [0.0f32, 0.25, 1.0, 1.5, -0.5] {
                src.extend_from_slice(&[v, v, v, a]);
            }
        }
        let n = src.len() / 4;
        let mut out = vec![0f32; n * 3];
        converter(T::F32, transfer).convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut out),
            n as u32,
        );
        for (i, p) in src.as_chunks::<4>().0.iter().enumerate() {
            let want = oetf(transfer, blend(eotf(transfer, p[0]), p[3], m[0]));
            let got = out[i * 3];
            assert!(
                (got - want).abs() <= 1e-5 + want.abs() * 1e-5,
                "{transfer:?} {p:?}: {got} {want}"
            );
        }
    }
}
