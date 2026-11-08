//! Recheck the PR116 Display-P3 to named-sRGB witness on MoxCMS 0.9.2.
//! The f64 profile-curve/colorant oracle and seed are retained from PR116.
#![forbid(unsafe_code)]
use zenpixels::{Cicp, ColorProfileSource as Source, PixelFormat as F};
use zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
use zenpixels_convert::{ConvertOptions, MoxCms, PluggableCms, RowTransformMut};
const SRGB: Source<'static> = Source::Cicp(Cicp::new(1, 13, 0, true));
fn build(src: Source<'_>, dst: Source<'_>, from: F, to: F) -> Box<dyn RowTransformMut> {
    MoxCms
        .build_source_transform(src, dst, from, to, &ConvertOptions::permissive())
        .unwrap()
        .unwrap()
}
fn main() {
    // Exact math on the profile's curve and colorants, into sRGB scaled to the PCS white.
    let p3 = moxcms::ColorProfile::new_from_slice(DISPLAY_P3_V4).unwrap();
    let srgb = moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: moxcms::CicpColorPrimaries::Bt709,
        transfer_characteristics: moxcms::TransferCharacteristics::Srgb,
        matrix_coefficients: moxcms::MatrixCoefficients::Identity,
        full_range: true,
    });
    let (r, g, b) = (srgb.red_colorant, srgb.green_colorant, srgb.blue_colorant);
    let to_xyz = [
        [r.x, g.x, b.x].map(|v| v * 0.9642 / (r.x + g.x + b.x)),
        [r.y, g.y, b.y].map(|v| v / (r.y + g.y + b.y)),
        [r.z, g.z, b.z].map(|v| v * 0.8249 / (r.z + g.z + b.z)),
    ];
    let matrix = moxcms::Matrix3d { v: to_xyz }
        .inverse()
        .mat_mul(p3.colorant_matrix())
        .v;
    let moxcms::ToneReprCurve::Parametric(curve) = p3.red_trc.as_ref().unwrap() else {
        panic!("Display P3 stores a parametric curve")
    };
    let c: Vec<f64> = curve.iter().map(|&v| f64::from(v)).collect();
    let decode = |v: f64| {
        if v >= c[4] {
            (c[1] * v + c[2]).powf(c[0])
        } else {
            c[3] * v
        }
    };
    let encode = |v: f64| {
        if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };

    let mut state = 0x9E37_79B9u32;
    let input: Vec<u16> = (0..3 * 4096)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 16) as u16
        })
        .collect();
    let mut out = vec![0u16; input.len()];
    build(Source::Icc(DISPLAY_P3_V4), SRGB, F::Rgb16, F::Rgb16)
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut out),
            4096,
        )
        .unwrap();

    let floats: Vec<f32> = input.iter().map(|&v| f32::from(v) / 65535.0).collect();
    let mut floating = vec![0.0f32; input.len()];
    build(Source::Icc(DISPLAY_P3_V4), SRGB, F::RgbF32, F::RgbF32)
        .try_transform_row(
            bytemuck::cast_slice(&floats),
            bytemuck::cast_slice_mut(&mut floating),
            4096,
        )
        .unwrap();
    let rounded: Vec<u16> = floating
        .iter()
        .map(|v| (f64::from(v.clamp(0.0, 1.0)) * 65535.0).round() as u16)
        .collect();
    for (name, out) in [
        ("production-mox092-native-u16", &out),
        ("production-mox092-f32-rounded", &rounded),
    ] {
        let (mut signed, mut wrong, mut worst) = (0.0f64, 0usize, 0.0f64);
        for (from, to) in input.as_chunks::<3>().0.iter().zip(out.as_chunks::<3>().0) {
            let linear = from.map(|v| decode(f64::from(v) / 65535.0));
            for (row, got) in matrix.iter().zip(to) {
                let value = row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
                let exact = encode(value.clamp(0.0, 1.0)) * 65535.0;
                let error = f64::from(*got) - exact;
                signed += error;
                worst = worst.max(error.abs());
                wrong += usize::from(f64::from(*got) != exact.round());
            }
        }
        let share = wrong as f64 / out.len() as f64;
        let bias = signed / out.len() as f64;
        println!(
            "{name}: components={} missed_nearest={wrong} share={share:.17} mean_signed={bias:.17} max_abs={worst:.17}",
            out.len()
        );
    }
}
