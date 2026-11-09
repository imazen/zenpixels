//! PQ profiles: bounded conventional LUTs and float pipelines in both directions.
//! ICC.1:2022 8.10, 10.12/13/16; H.273 Table 3. Build/test code only.
use super::alloc::{vec, vec::Vec};
use lcms::{gamma, lut, profile, tagtypes::TagObject};
use moxcms::{
    ColorProfile, LutMultidimensionalType, LutStore, LutWarehouse, Matrix3d, ToneReprCurve,
    Vector3d,
};

fn decode(x: f64) -> f64 {
    let p = x.powf(32.0 / 2523.0);
    ((p - 107.0 / 128.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * p)).powf(8192.0 / 1305.0)
}
fn encode_sample(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let p = x.powf(1305.0 / 8192.0);
    ((107.0 / 128.0 + 2413.0 / 128.0 * p) / (1.0 + 2392.0 / 128.0 * p)).powf(2523.0 / 32.0)
}

// Separate the exact-binary affine term from the reciprocal. Combining them
// into two independently rounded large coefficients loses precision at white.
fn formula(parameters: [f64; 4], bounds: Option<(f32, f32)>) -> Option<gamma::ToneCurve> {
    let mut params = [0.0; 10];
    params[..4].copy_from_slice(&parameters);
    let (lower, upper) = bounds.unwrap_or((-1e22, 1e22));
    let mut segments = Vec::new();
    if bounds.is_some() {
        segments.push(gamma::CurveSegment {
            x0: -1e22,
            x1: lower,
            r#type: 6,
            params: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        });
    }
    segments.push(gamma::CurveSegment {
        x0: lower,
        x1: upper,
        r#type: 6,
        params,
        ..Default::default()
    });
    if bounds.is_some() {
        segments.push(gamma::CurveSegment {
            x0: upper,
            x1: 1e22,
            r#type: 6,
            params: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        });
    }
    gamma::build_segmented_tone_curve(None, &segments)
}

fn white_scale(white: f32) -> f64 {
    let mut scale = (1.0 / f64::from(white)) as f32;
    if f64::from(scale) * f64::from(white) > 1.0 {
        scale = f32::from_bits(scale.to_bits() - 1);
    }
    f64::from(scale)
}

fn float_curves(inverse: bool) -> Option<Vec<gamma::ToneCurve>> {
    let (c1, c2, c3) = (107.0 / 128.0, 2413.0 / 128.0, 2392.0 / 128.0);
    if inverse {
        let a = (c2 / c3) as f32;
        let e = (c2 / c3 - c1) as f32;
        let black = a - e;
        let reciprocal_white = (1.0 / (1.0 + c3)) as f32;
        let white = (f64::from(a) - f64::from(e) * f64::from(reciprocal_white)) as f32;
        Some(vec![
            formula([1305.0 / 8192.0, 1.0, 0.0, 0.0], Some((0.0, 1.0)))?,
            formula([-1.0, c3, 1.0, 0.0], None)?,
            formula([1.0, -f64::from(e), f64::from(a), 0.0], None)?,
            formula(
                [2523.0 / 32.0, white_scale(white), 0.0, 0.0],
                Some((black, white)),
            )?,
        ])
    } else {
        let multiplier = (c3 / (c2 - c1 * c3)) as f32;
        let offset = (-1.0 / c3) as f32;
        let white = (1.0 / (f64::from(multiplier) * (c2 - c3)) + f64::from(offset)) as f32;
        Some(vec![
            formula([32.0 / 2523.0, 1.0, 0.0, 0.0], Some((0.0, 1.0)))?,
            formula([1.0, -c3, c2, 0.0], None)?,
            formula([-1.0, f64::from(multiplier), 0.0, f64::from(offset)], None)?,
            formula(
                [8192.0 / 1305.0, white_scale(white), 0.0, 0.0],
                Some((0.0, white)),
            )?,
        ])
    }
}

fn identity_clut(inputs: usize, outputs: usize) -> (Vec<u16>, [u8; 16]) {
    let mut grid = [0; 16];
    grid[..inputs].fill(2);
    let mut values = Vec::new();
    for point in 0..1 << inputs {
        for channel in 0..outputs {
            let input = if inputs == 1 {
                0
            } else if outputs == 1 {
                1
            } else {
                channel
            };
            values.push(if point & (1 << (inputs - input - 1)) == 0 {
                0
            } else {
                65535
            });
        }
    }
    (values, grid)
}

pub(super) fn encode(mut source: ColorProfile, gray: bool) -> Option<Vec<u8>> {
    let channels = if gray { 1 } else { 3 };
    // Match the actual stored colorants, including their s15Fixed16 rounding.
    // A different float matrix would describe a different linear working gamut.
    let stored = ColorProfile::new_from_slice(&source.encode().ok()?).ok()?;
    let matrix = if gray {
        Matrix3d {
            v: [[0.9642, 0.0, 0.0], [1.0, 0.0, 0.0], [0.8249, 0.0, 0.0]],
        }
    } else {
        Matrix3d {
            v: [
                [
                    stored.red_colorant.x,
                    stored.green_colorant.x,
                    stored.blue_colorant.x,
                ],
                [
                    stored.red_colorant.y,
                    stored.green_colorant.y,
                    stored.blue_colorant.y,
                ],
                [
                    stored.red_colorant.z,
                    stored.green_colorant.z,
                    stored.blue_colorant.z,
                ],
            ],
        }
    };
    // Conventional PCSXYZ is u1Fixed15, i.e. normalized by 65535/32768.
    let factor = 65535.0 / 32768.0;
    let root = ToneReprCurve::Lut(
        (0..=4096)
            .map(|i| (decode(f64::from(i) / 4096.0).powf(0.125) * 65535.0).round() as u16)
            .collect(),
    );
    let unroot = ToneReprCurve::Lut(
        (0..=4096)
            .map(|i| (encode_sample((f64::from(i) / 4096.0).powi(8)) * 65535.0).round() as u16)
            .collect(),
    );
    let identity = ToneReprCurve::Parametric(vec![1.0]);
    let (clut, grid) = identity_clut(channels, 3);
    let forward = LutWarehouse::Multidimensional(LutMultidimensionalType {
        num_input_channels: channels as u8,
        num_output_channels: 3,
        grid_points: grid,
        clut: Some(LutStore::Store16(clut)),
        a_curves: vec![root; channels],
        m_curves: vec![ToneReprCurve::Parametric(vec![8.0]); 3],
        matrix: Matrix3d {
            v: matrix.v.map(|r| r.map(|v| v / factor)),
        },
        bias: Vector3d { v: [0.0; 3] },
        b_curves: vec![identity.clone(); 3],
    });
    let inverse = if gray {
        Matrix3d::IDENTITY
    } else {
        matrix.inverse()
    };
    let (clut, grid) = identity_clut(3, channels);
    let backward = LutWarehouse::Multidimensional(LutMultidimensionalType {
        num_input_channels: 3,
        num_output_channels: channels as u8,
        grid_points: grid,
        clut: Some(LutStore::Store16(clut)),
        a_curves: vec![unroot; channels],
        m_curves: vec![ToneReprCurve::Parametric(vec![0.125]); 3],
        matrix: Matrix3d {
            v: inverse.v.map(|r| r.map(|v| v * factor)),
        },
        bias: Vector3d { v: [0.0; 3] },
        b_curves: vec![identity; 3],
    });
    source.lut_a_to_b_perceptual = Some(forward.clone());
    source.lut_a_to_b_colorimetric = Some(forward.clone());
    source.lut_a_to_b_saturation = Some(forward);
    source.lut_b_to_a_perceptual = Some(backward.clone());
    source.lut_b_to_a_colorimetric = Some(backward.clone());
    source.lut_b_to_a_saturation = Some(backward);
    let encoded = source.encode().ok()?;
    let mut output = profile::open_profile_from_mem(None, &encoded)?;
    let decoding = float_curves(false)?;
    let encoding = float_curves(true)?;
    let mut forward = lut::pipeline_alloc(None, channels as u32, 3)?;
    for curve in &decoding {
        if !forward.insert_stage(
            lut::StageLoc::AtEnd,
            lut::stage_alloc_tone_curves(None, channels as u32, Some(&vec![curve; channels]))?,
        ) {
            return None;
        }
    }
    let coefficients: Vec<_> = if gray {
        vec![0.9642, 1.0, 0.8249]
    } else {
        matrix.v.into_iter().flatten().collect()
    };
    if !forward.insert_stage(
        lut::StageLoc::AtEnd,
        lut::stage_alloc_matrix(None, 3, channels as u32, &coefficients, None)?,
    ) {
        return None;
    }
    let mut backward = lut::pipeline_alloc(None, 3, channels as u32)?;
    let coefficients: Vec<_> = if gray {
        vec![0.0, 1.0, 0.0]
    } else {
        inverse.v.into_iter().flatten().collect()
    };
    if !backward.insert_stage(
        lut::StageLoc::AtEnd,
        lut::stage_alloc_matrix(None, channels as u32, 3, &coefficients, None)?,
    ) {
        return None;
    }
    for curve in &encoding {
        if !backward.insert_stage(
            lut::StageLoc::AtEnd,
            lut::stage_alloc_tone_curves(None, channels as u32, Some(&vec![curve; channels]))?,
        ) {
            return None;
        }
    }
    // Absolute uses the conventional colorimetric transform and the profile's
    // chromatic adaptation; no independently defined absolute MPE is invented.
    for intent in *b"012" {
        for (prefix, pipeline) in [(b'D', &forward), (b'B', &backward)] {
            let signature = if prefix == b'D' {
                [b'D', b'2', b'B', intent]
            } else {
                [b'B', b'2', b'D', intent]
            };
            if !profile::cms_write_tag(
                &mut output,
                u32::from_be_bytes(signature),
                Some(TagObject::Mpe(pipeline.clone())),
            ) {
                return None;
            }
        }
    }
    let mut bytes = profile::save_profile_to_mem(&mut output)?;
    bytes[8..12].copy_from_slice(&[4, 0x40, 0, 0]);
    Some(bytes)
}
