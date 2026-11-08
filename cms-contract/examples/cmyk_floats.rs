//! Existing JXL CMYK regression samples through the pinned Rust LCMS adapter.
use cms_contract::LcmsPort;
use zenpixels::{Cicp, ColorProfileSource, PixelFormat};
use zenpixels_convert::{cms::PluggableCms, policy::ConvertOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("expected CMYK ICC profile")?;
    let icc = std::fs::read(path)?;
    let mut transform = LcmsPort
        .build_source_transform(
            ColorProfileSource::Icc(&icc),
            ColorProfileSource::Cicp(Cicp::new(1, 13, 0, true)),
            PixelFormat::CmykF32Inverted,
            PixelFormat::RgbF32,
            &ConvertOptions::permissive().with_clip_out_of_gamut(false),
        )
        .ok_or("backend declined")?
        .map_err(|e| format!("{e:?}"))?;
    for sample in [
        [0.2863f32, 1.0, 0.6706, 1.0],
        [0.2347, 0.9601, 0.5918, 0.9813],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ] {
        let mut rgb = [0.0f32; 3];
        transform
            .try_transform_row(
                bytemuck::cast_slice(&sample),
                bytemuck::cast_slice_mut(&mut rgb),
                1,
            )
            .map_err(|e| format!("{e:?}"))?;
        println!("rust-lcms\t{sample:?}\t{rgb:?}");
    }
    Ok(())
}
