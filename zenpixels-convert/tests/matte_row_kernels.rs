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

/// The background in `transfer`'s linear, at the default reference white.
fn matte(transfer: Tf) -> [f32; 3] {
    let white = match transfer {
        Tf::Pq => 203.0 / 10_000.0,
        Tf::Hlg => 0.203f32.powf(1.0 / 1.2),
        _ => 1.0,
    };
    [MATTE.0, MATTE.1, MATTE.2].map(|c| ds::srgb_u8_to_linear(c) * white)
}

fn converter(t: T, transfer: Tf) -> RowConverter {
    let from = D::new(t, L::Rgba, Some(AlphaMode::Straight), transfer);
    let to = D::new(t, L::Rgb, None, transfer);
    let policy = AlphaPolicy::CompositeOnto {
        background: zenpixels::Color::srgb8(MATTE.0, MATTE.1, MATTE.2),
        reference_white: None,
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
    let m = matte(Tf::Srgb);
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

/// Every gray source, alpha and matte value. A random sample at a fixed matte misses the 32
/// combinations where a correctly rounded alpha would change the result.
#[test]
fn srgb_u8_gray_matches_its_tables_exhaustively() {
    let src: Vec<u8> = (0..=255u8)
        .flat_map(|g| (0..=255u8).flat_map(move |a| [g, g, g, a]))
        .collect();
    let mut out = vec![0u8; 65536 * 3];
    let opts = |m: u8| {
        ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
            background: zenpixels::Color::srgb8(m, m, m),
            reference_white: None,
        })
    };
    let (from, to) = (
        D::new(T::U8, L::Rgba, Some(AlphaMode::Straight), Tf::Srgb),
        D::new(T::U8, L::Rgb, None, Tf::Srgb),
    );
    for m in 0..=255u8 {
        RowConverter::new_explicit(from, to, &opts(m))
            .unwrap()
            .convert_row(&src, &mut out, 65536);
        let matte = ds::srgb_u8_to_linear(m);
        for (p, o) in src.as_chunks::<4>().0.iter().zip(out.as_chunks::<3>().0) {
            let a = f32::from(p[3]) * (1.0 / 255.0);
            let want = ds::linear_to_srgb_u8(blend(ds::srgb_u8_to_linear(p[0]), a, matte));
            assert_eq!(o[0], want, "gray {} alpha {} matte {m}", p[0], p[3]);
        }
    }
}

#[test]
fn integer_output_stays_within_one_code() {
    for transfer in CURVES {
        let m = matte(transfer);
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
    for transfer in CURVES {
        let m = matte(transfer);
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
    for transfer in CURVES {
        let m = matte(transfer);
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
    for transfer in CURVES {
        let m = matte(transfer);
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

mod reference_white {
    use super::*;
    use zenpixels::hdr::DiffuseWhite;

    /// ST 2084 EOTF in cd/m².
    fn pq_nits(signal: f64) -> f64 {
        let (m1, m2) = (2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0);
        let (c1, c2, c3) = (
            3424.0 / 4096.0,
            2413.0 / 4096.0 * 32.0,
            2392.0 / 4096.0 * 32.0,
        );
        let p = signal.powf(1.0 / m2);
        10_000.0 * (f64::max(p - c1, 0.0) / (c2 - c3 * p)).powf(1.0 / m1)
    }

    /// `pixel` composited onto white, as one RGB sample.
    fn over_white(transfer: Tf, pixel: [f32; 4], reference_white: Option<DiffuseWhite>) -> f32 {
        let policy = AlphaPolicy::CompositeOnto {
            background: zenpixels::Color::srgb8(255, 255, 255),
            reference_white,
        };
        let options = ConvertOptions::permissive().with_alpha_policy(policy);
        let from = D::new(T::F32, L::Rgba, Some(AlphaMode::Straight), transfer);
        let to = D::new(T::F32, L::Rgb, None, transfer);
        let mut out = [0f32; 3];
        RowConverter::new_explicit(from, to, &options)
            .unwrap()
            .convert_row(
                bytemuck::cast_slice(&pixel),
                bytemuck::cast_slice_mut(&mut out),
                1,
            );
        out[0]
    }

    const TRANSPARENT: [f32; 4] = [0.0; 4];

    #[test]
    fn pq_background_defaults_to_bt2408() {
        let nits = pq_nits(f64::from(over_white(Tf::Pq, TRANSPARENT, None)));
        assert!((nits - 203.0).abs() < 0.1, "{nits}");
    }

    #[test]
    fn pq_background_uses_the_callers_reference_white() {
        for want in [100.0f32, 203.0, 1000.0] {
            let signal = over_white(Tf::Pq, TRANSPARENT, Some(DiffuseWhite::new(want)));
            let nits = pq_nits(f64::from(signal));
            assert!(
                (nits - f64::from(want)).abs() < f64::from(want) * 5e-4,
                "{want}: {nits}"
            );
        }
    }

    /// BT.2408: HLG reference white is at 75% signal.
    #[test]
    fn hlg_background_defaults_to_three_quarters_signal() {
        let signal = over_white(Tf::Hlg, TRANSPARENT, None);
        assert!((signal - 0.75).abs() < 0.002, "{signal}");
    }

    #[test]
    fn hlg_background_follows_the_callers_reference_white() {
        let low = over_white(Tf::Hlg, TRANSPARENT, Some(DiffuseWhite::new(100.0)));
        let high = over_white(Tf::Hlg, TRANSPARENT, Some(DiffuseWhite::new(400.0)));
        assert!(low < 0.75 && high > 0.75, "{low} {high}");
    }

    /// Opaque pixels show no background, so reference white can't change them.
    #[test]
    fn opaque_pixels_ignore_reference_white() {
        for transfer in [Tf::Pq, Tf::Hlg] {
            let pixel = [0.25, 0.25, 0.25, 1.0];
            let a = over_white(transfer, pixel, Some(DiffuseWhite::new(100.0)));
            let b = over_white(transfer, pixel, Some(DiffuseWhite::new(1000.0)));
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// SDR white is the format's white, whatever reference white says.
    #[test]
    fn sdr_sources_ignore_reference_white() {
        for transfer in [Tf::Srgb, Tf::Bt709, Tf::Gamma22, Tf::Linear] {
            let a = over_white(transfer, TRANSPARENT, None);
            let b = over_white(transfer, TRANSPARENT, Some(DiffuseWhite::new(1000.0)));
            assert_eq!(a.to_bits(), b.to_bits(), "{transfer:?}");
            assert!((a - 1.0).abs() < 2e-6, "{transfer:?}: {a}");
        }
    }
}
