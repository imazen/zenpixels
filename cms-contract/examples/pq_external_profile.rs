//! Inspect an external PQ ICC without treating it as a replacement fixture.
use zenpixels::{ColorProfileSource as Source, PixelFormat};
use zenpixels_convert::{
    cms::PluggableCms, cms_lcms::Lcms, cms_moxcms::MoxCms, policy::ConvertOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).ok_or("expected ICC file")?;
    let icc = std::fs::read(path)?;
    let mut linear = moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: moxcms::CicpColorPrimaries::Bt2020,
        transfer_characteristics: moxcms::TransferCharacteristics::Linear,
        matrix_coefficients: moxcms::MatrixCoefficients::Identity,
        full_range: true,
    });
    linear.update_rgb_colorimetry(
        moxcms::XyY {
            x: 0.3127,
            y: 0.3290,
            yb: 1.0,
        },
        moxcms::ColorPrimaries {
            red: moxcms::Chromaticity { x: 0.708, y: 0.292 },
            green: moxcms::Chromaticity { x: 0.170, y: 0.797 },
            blue: moxcms::Chromaticity { x: 0.131, y: 0.046 },
        },
    );
    let linear = linear.encode()?;
    let codes = [0.0f32, 0.0001, 0.001, 0.01, 0.1, 0.18, 0.5, 0.75, 0.9, 1.0];
    let input: Vec<_> = codes.into_iter().flat_map(|v| [v; 3]).collect();
    // Separate the backend's own result from the zenpixels row adapter.
    let mut raw_source = moxcms::ColorProfile::new_from_slice(&icc)?;
    let mut raw_target = moxcms::ColorProfile::new_from_slice(&linear)?;
    raw_source.cicp = None;
    raw_target.cicp = None;
    let raw = raw_source.create_transform_f32(
        moxcms::Layout::Rgb,
        &raw_target,
        moxcms::Layout::Rgb,
        moxcms::TransformOptions {
            rendering_intent: moxcms::RenderingIntent::Perceptual,
            allow_use_cicp_transfer: false,
            prefer_fixed_point: false,
            barycentric_weight_scale: moxcms::BarycentricWeightScale::High,
            interpolation_method: moxcms::InterpolationMethod::Tetrahedral,
            allow_extended_range_rgb_xyz: true,
        },
    )?;
    let mut raw_output = vec![0.0f32; input.len()];
    raw.transform(&input, &mut raw_output)?;
    for (code, rgb) in codes.into_iter().zip(raw_output.as_chunks::<3>().0) {
        println!("raw-mox\tcode={code:.9}\tlinear_rgb={rgb:?}");
    }
    for (name, cms) in [
        ("mox", &MoxCms as &dyn PluggableCms),
        ("lcms", &Lcms as &dyn PluggableCms),
    ] {
        let result = cms.build_source_transform(
            Source::Icc(&icc),
            Source::Icc(&linear),
            PixelFormat::RgbF32,
            PixelFormat::RgbF32,
            &ConvertOptions::permissive().with_clip_out_of_gamut(false),
        );
        let Some(result) = result else {
            println!("{name}\tdeclined");
            continue;
        };
        let mut transform = result.map_err(|e| format!("{e:?}"))?;
        let mut output = vec![0.0f32; input.len()];
        transform
            .try_transform_row(
                bytemuck::cast_slice(&input),
                bytemuck::cast_slice_mut(&mut output),
                codes.len() as u32,
            )
            .map_err(|e| format!("{e:?}"))?;
        for (code, rgb) in codes.into_iter().zip(output.as_chunks::<3>().0) {
            let e = f64::from(code).powf(32.0 / 2523.0);
            let pq = ((e - 107.0 / 128.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * e))
                .powf(8192.0 / 1305.0);
            println!("{name}\tcode={code:.9}\tpq_normalized={pq:.12e}\tlinear_rgb={rgb:?}");
        }
    }
    Ok(())
}
