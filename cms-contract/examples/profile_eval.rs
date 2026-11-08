//! Evaluate exact ICC bytes, arbitrary float rows and a literal rendering intent.
#![forbid(unsafe_code)]
use lcms::{profile, types, xform};
use zenpixels::{ColorProfileSource, PixelFormat};
use zenpixels_convert::{cms::PluggableCms, cms_lcms::Lcms, policy::ConvertOptions};

fn format(icc: &[u8]) -> Result<(u32, PixelFormat, usize), Box<dyn std::error::Error>> {
    match icc.get(16..20) {
        Some(b"RGB ") => Ok(((1 << 22) | (4 << 16) | (3 << 3) | 4, PixelFormat::RgbF32, 3)),
        Some(b"GRAY") => Ok((
            (1 << 22) | (3 << 16) | (1 << 3) | 4,
            PixelFormat::GrayF32,
            1,
        )),
        Some(b"XYZ ") => Ok(((1 << 22) | (9 << 16) | (3 << 3) | 4, PixelFormat::RgbF32, 3)),
        _ => Err("only RGB/GRAY/XYZ float fixtures supported".into()),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("expected SOURCE.icc DEST.icc INPUT.f32 OUTPUT.f32 INTENT|production".into());
    }
    let src = std::fs::read(&args[0])?;
    let dst = std::fs::read(&args[1])?;
    let (src_code, src_format, src_channels) = format(&src)?;
    let (dst_code, dst_format, dst_channels) = format(&dst)?;
    let input = std::fs::read(&args[2])?;
    if input.len() % (src_channels * 4) != 0 {
        return Err("partial input pixel".into());
    }
    let count = input.len() / (src_channels * 4);
    let mut output = vec![0u8; count * dst_channels * 4];
    if args[4] == "production" {
        let mut transform = Lcms
            .build_source_transform(
                ColorProfileSource::Icc(&src),
                ColorProfileSource::Icc(&dst),
                src_format,
                dst_format,
                &ConvertOptions::permissive().with_clip_out_of_gamut(false),
            )
            .ok_or("production transform declined")?
            .map_err(|e| format!("{e:?}"))?;
        transform
            .try_transform_row(&input, &mut output, u32::try_from(count)?)
            .map_err(|e| format!("{e:?}"))?;
    } else {
        let intent: u32 = args[4].parse()?;
        if intent > 3 {
            return Err("intent outside 0..3".into());
        }
        let mut profiles = [
            profile::open_profile_from_mem(None, &src).ok_or("source parse")?,
            profile::open_profile_from_mem(None, &dst).ok_or("destination parse")?,
        ];
        let adaptation = if intent == 3 { 0.0 } else { 1.0 };
        let mut transform = xform::create_extended_transform(
            None,
            &mut profiles,
            &mut [false; 2],
            &[intent; 2],
            &[adaptation; 2],
            None,
            0,
            src_code,
            dst_code,
            types::FLAGS_NOOPTIMIZE | types::FLAGS_NOCACHE,
        )
        .ok_or("literal-intent transform")?;
        transform.do_transform(&input, &mut output, u32::try_from(count)?);
    }
    if args[3] == "-" {
        use std::io::Write;
        std::io::stdout().write_all(&output)?;
    } else {
        std::fs::write(&args[3], output)?;
    }
    Ok(())
}
