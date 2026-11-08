//! Characterize PQ matrix/TRC, compound mAB and float MPE representations.
//! Outputs are measurements, not replacement acceptance thresholds.
use lcms::{gamma, lut, profile, tagtypes::TagObject};
use moxcms::{
    ColorProfile, LutMultidimensionalType, LutStore, LutWarehouse, Matrix3d, ToneReprCurve,
    Vector3d,
};
use zenpixels::{ColorProfileSource, PixelFormat};
use zenpixels_convert::{
    cms::PluggableCms, cms_lcms::Lcms, cms_moxcms::MoxCms, policy::ConvertOptions,
};

fn pq(x: f64) -> f64 {
    let e = x.powf(32.0 / 2523.0);
    ((e - 107.0 / 128.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * e)).powf(8192.0 / 1305.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("expected RGB-PQ.icc GRAY-PQ.icc FRESH_OUTPUT_DIR".into());
    }
    let output = std::path::Path::new(&args[2]);
    std::fs::create_dir(output)?;
    for (kind, path) in [("rgb", &args[0]), ("gray", &args[1])] {
        let source = std::fs::read(path)?;
        let mut base = ColorProfile::new_from_slice(&source)?;
        let gray = kind == "gray";
        let inputs = if gray { 1 } else { 3 };
        let d50 = moxcms::Xyzd {
            x: 0.9642,
            y: 1.0,
            z: 0.8249,
        };
        base.media_white_point = Some(d50);
        let colorants = if gray {
            [d50.x, 0.0, 0.0, d50.y, 0.0, 0.0, d50.z, 0.0, 0.0]
        } else {
            [
                base.red_colorant.x,
                base.green_colorant.x,
                base.blue_colorant.x,
                base.red_colorant.y,
                base.green_colorant.y,
                base.blue_colorant.y,
                base.red_colorant.z,
                base.green_colorant.z,
                base.blue_colorant.z,
            ]
        };
        let mut dest = if gray {
            ColorProfile::new_srgb()
        } else {
            base.clone()
        };
        let linear = ToneReprCurve::Parametric(vec![1.0]);
        dest.red_trc = Some(linear.clone());
        dest.green_trc = Some(linear.clone());
        dest.blue_trc = Some(linear);
        dest.cicp = None;
        let dest = dest.encode()?;
        let initial = base.encode()?;
        let mut variants = vec![("original", initial.clone())];
        for power in [3.0f64, 8.0] {
            let mut shaped = base.clone();
            let root = ToneReprCurve::Lut(
                (0..32767)
                    .map(|code| {
                        (pq(f64::from(code) / 32766.0).powf(1.0 / power) * 65535.0).round() as u16
                    })
                    .collect(),
            );
            let mut grid = [0u8; 16];
            grid[..inputs].fill(2);
            let mut clut = Vec::new();
            if gray {
                clut.extend([0, 0, 0, 65535, 65535, 65535]);
            } else {
                for r in [0, 65535] {
                    for g in [0, 65535] {
                        for b in [0, 65535] {
                            clut.extend([r, g, b]);
                        }
                    }
                }
            }
            let factor = 65535.0 / 32768.0;
            let matrix = if gray {
                Matrix3d {
                    v: [
                        [d50.x / factor, 0.0, 0.0],
                        [d50.y / factor, 0.0, 0.0],
                        [d50.z / factor, 0.0, 0.0],
                    ],
                }
            } else {
                Matrix3d {
                    v: core::array::from_fn(|i| {
                        core::array::from_fn(|j| colorants[3 * i + j] / factor)
                    }),
                }
            };
            let table = LutWarehouse::Multidimensional(LutMultidimensionalType {
                num_input_channels: inputs as u8,
                num_output_channels: 3,
                grid_points: grid,
                clut: Some(LutStore::Store16(clut)),
                a_curves: vec![root; inputs],
                m_curves: vec![ToneReprCurve::Parametric(vec![power as f32]); 3],
                matrix,
                bias: Vector3d { v: [0.0; 3] },
                b_curves: vec![ToneReprCurve::Parametric(vec![1.0]); 3],
            });
            shaped.lut_a_to_b_colorimetric = Some(table.clone());
            shaped.lut_a_to_b_perceptual = Some(table);
            shaped.cicp = None;
            variants.push((if power == 3.0 { "mab3" } else { "mab8" }, shaped.encode()?));
        }
        let mut mpe = profile::open_profile_from_mem(None, &initial).ok_or("open MPE source")?;
        let samples: Vec<_> = (0..=65535)
            .map(|c| pq(f64::from(c) / 65535.0) as f32)
            .collect();
        let curve = gamma::build_tabulated_tone_curve_float(None, &samples).ok_or("float curve")?;
        let mut pipe = lut::pipeline_alloc(None, inputs as u32, 3).ok_or("pipeline")?;
        assert!(
            pipe.insert_stage(
                lut::StageLoc::AtEnd,
                lut::stage_alloc_tone_curves(None, inputs as u32, Some(&vec![&curve; inputs]))
                    .ok_or("curves stage")?
            )
        );
        let matrix = if gray {
            vec![d50.x, d50.y, d50.z]
        } else {
            colorants.to_vec()
        };
        assert!(pipe.insert_stage(
            lut::StageLoc::AtEnd,
            lut::stage_alloc_matrix(None, 3, inputs as u32, &matrix, None).ok_or("matrix stage")?
        ));
        for signature in [*b"D2B0", *b"D2B1"] {
            assert!(profile::cms_write_tag(
                &mut mpe,
                u32::from_be_bytes(signature),
                Some(TagObject::Mpe(pipe.clone()))
            ));
        }
        variants.push((
            "mpe",
            profile::save_profile_to_mem(&mut mpe).ok_or("save MPE")?,
        ));
        let mut formula =
            profile::open_profile_from_mem(None, &initial).ok_or("open formula profile")?;
        let mut pipe = lut::pipeline_alloc(None, inputs as u32, 3).ok_or("formula pipeline")?;
        let divisor = (2413.0 / 128.0) / (2392.0 / 128.0) - 107.0 / 128.0;
        for parameters in [
            [32.0 / 2523.0, 1.0, 0.0, 0.0],
            [
                -1.0,
                -(2392.0 / 128.0) / divisor,
                (2413.0 / 128.0) / divisor,
                -128.0 / 2392.0,
            ],
            [8192.0 / 1305.0, 1.0, 0.0, 0.0],
        ] {
            let mut params = [0.0; 10];
            params[..4].copy_from_slice(&parameters);
            let segment = gamma::CurveSegment {
                x0: -1e22,
                x1: 1e22,
                r#type: 6,
                params,
                ..Default::default()
            };
            let curve =
                gamma::build_segmented_tone_curve(None, &[segment]).ok_or("formula curve")?;
            assert!(
                pipe.insert_stage(
                    lut::StageLoc::AtEnd,
                    lut::stage_alloc_tone_curves(None, inputs as u32, Some(&vec![&curve; inputs]))
                        .ok_or("formula curves stage")?
                )
            );
        }
        let matrix = if gray {
            vec![d50.x, d50.y, d50.z]
        } else {
            colorants.to_vec()
        };
        assert!(
            pipe.insert_stage(
                lut::StageLoc::AtEnd,
                lut::stage_alloc_matrix(None, 3, inputs as u32, &matrix, None)
                    .ok_or("formula matrix")?
            )
        );
        for signature in [*b"D2B0", *b"D2B1"] {
            assert!(profile::cms_write_tag(
                &mut formula,
                u32::from_be_bytes(signature),
                Some(TagObject::Mpe(pipe.clone()))
            ));
        }
        variants.push((
            "mpe-formula",
            profile::save_profile_to_mem(&mut formula).ok_or("save formula MPE")?,
        ));
        for (variant, bytes) in variants {
            std::fs::write(output.join(format!("{kind}-{variant}.icc")), &bytes)?;
            let fmt = if gray {
                PixelFormat::GrayF32
            } else {
                PixelFormat::RgbF32
            };
            let input: Vec<f32> = (0..=65535)
                .flat_map(|c| vec![c as f32 / 65535.0; inputs])
                .collect();
            for (engine, cms) in [
                ("mox", &MoxCms as &dyn PluggableCms),
                ("lcms", &Lcms as &dyn PluggableCms),
            ] {
                let result = cms.build_source_transform(
                    ColorProfileSource::Icc(&bytes),
                    ColorProfileSource::Icc(&dest),
                    fmt,
                    PixelFormat::RgbF32,
                    &ConvertOptions::permissive().with_clip_out_of_gamut(false),
                );
                let Some(Ok(mut transform)) = result else {
                    println!("{kind}\t{variant}\t{engine}\tunsupported");
                    continue;
                };
                let mut values = vec![0.0f32; 65536 * 3];
                transform
                    .try_transform_row(
                        bytemuck::cast_slice(&input),
                        bytemuck::cast_slice_mut(&mut values),
                        65536,
                    )
                    .map_err(|e| format!("{e:?}"))?;
                let (mut max_abs, mut max_rel, mut zeros) = (0.0f64, 0.0f64, 0usize);
                for c in 1..=65535 {
                    let expected = pq(f64::from(input[c * inputs]));
                    let actual = f64::from(values[c * 3]);
                    max_abs = max_abs.max((actual - expected).abs());
                    if expected >= 1e-7 {
                        max_rel = max_rel.max((actual / expected - 1.0).abs());
                    }
                    zeros += usize::from(actual == 0.0 && expected > 0.0);
                }
                println!(
                    "{kind}\t{variant}\t{engine}\tmax_abs={max_abs:.9e}\trel_above_0.001nit={max_rel:.9e}\tzeros={zeros}\twhite={:?}",
                    &values[65535 * 3..]
                );
            }
        }
    }
    Ok(())
}
