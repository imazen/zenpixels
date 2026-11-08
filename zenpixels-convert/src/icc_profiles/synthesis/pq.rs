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

// Geometric breakpoints retain the low PQ codes without a huge uniform LUT.
// Every segment is read back and checked independently by the qualification tools.
fn float_curve(inverse: bool) -> Option<gamma::ToneCurve> {
    let evaluate: fn(f64) -> f64 = if inverse { encode_sample } else { decode };
    let lower = if inverse { -48 } else { -16 };
    let mut segments = vec![gamma::CurveSegment {
        x0: -1e22,
        x1: 0.0,
        r#type: 6,
        params: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ..Default::default()
    }];
    let mut start = 0.0;
    for exponent in lower..=0 {
        let end = 2_f64.powi(exponent);
        let intervals = if inverse {
            256
        } else {
            ((end - start) * 65536.0).round().clamp(1.0, 512.0) as usize
        };
        let points = (0..=intervals)
            .map(|i| evaluate(start + (end - start) * i as f64 / intervals as f64) as f32)
            .collect();
        segments.push(gamma::CurveSegment {
            x0: start as f32,
            x1: end as f32,
            r#type: 0,
            n_grid_points: (intervals + 1) as u32,
            sampled_points: Some(points),
            ..Default::default()
        });
        start = end;
    }
    segments.push(gamma::CurveSegment {
        x0: 1.0,
        x1: 1e22,
        r#type: 6,
        params: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ..Default::default()
    });
    gamma::build_segmented_tone_curve(None, &segments)
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
    let decoding = float_curve(false)?;
    let encoding = float_curve(true)?;
    let mut forward = lut::pipeline_alloc(None, channels as u32, 3)?;
    if !forward.insert_stage(
        lut::StageLoc::AtEnd,
        lut::stage_alloc_tone_curves(None, channels as u32, Some(&vec![&decoding; channels]))?,
    ) {
        return None;
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
    if !backward.insert_stage(
        lut::StageLoc::AtEnd,
        lut::stage_alloc_tone_curves(None, channels as u32, Some(&vec![&encoding; channels]))?,
    ) {
        return None;
    }
    // Absolute uses the conventional colorimetric transform and the profile's
    // chromatic adaptation; no independently defined absolute MPE is invented.
    for intent in [b'0', b'1', b'2'] {
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
